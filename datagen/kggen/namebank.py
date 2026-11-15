"""A bank of realistic entity names, deconflicted once so a world never collides by accident.

Invented names (``kggen.names``) are capitalised made-up words, so any NER finds them and a keyword
search is flattered. The bank holds names that look like real ones: common first names, ordinary
surnames, English words for companies, places and events. They are deliberately unremarkable, so a
model cannot answer from memory ("who founded Cedar Works?") yet the text reads like text.

Every alias the generator derives from a name (``kggen.names.alias_forms``) is a function of one
word: a person's surname, an organisation's first word, a place's own name. So a collision between
two different entities is two names that share such a word. The bank rules it out before any
generation: every word in every pool is unique across all pools, first names included. Only the
shared endings the generator adds (``Holdings``, ``Summit`` and the like) repeat, as they do in
invented names. ``check_bank`` states the rules and ``tests`` run it.

A world draws names without replacement, so the bank bounds its size; ``CAPACITY`` says how many of
each type. It covers every preset up to ``large`` (10,000 entities); ``stress`` (50,000) needs
``name_style: invented``, and asking the bank for more than it holds is an error that says so.
"""

from __future__ import annotations

FIRST_NAMES: tuple[str, ...] = tuple(
    """
    Alma Brandon Camille Dario Elena Felix Greta Hugo Ingrid Jonas Keira Lucas Mireille Nikolai
    Odette Pavel Quinn Rosa Stefan Tessa Ulrich Vera Wendell Ximena Yusuf Zora Anselm Beatrix
    Cormac Delia Emil Fiona Gideon Hana Isaac Juno Kasper Lena Mateo Nadia Oskar Petra Rafael
    Selma Tobias Una Viktor Wanda Aiden Bianca Callum Dalia Evander Farah Gustav Helena Ivo
    Jasmin Kofi Linnea Marcel Noor Orla Prakash Renata Soren Talia Umar Valeria Willem Yara Zane
    Amara Bruno Clara Dmitri Esme Faisal Gemma Hector
    """.split()
)

SURNAMES: tuple[str, ...] = tuple(
    """
    Ashgrove Brandvold Castellan Dunmore Eskridge Fairweather Garrity Halvorsen Ingersoll Jarvinen
    Kettering Lindqvist Marchetti Nordahl Okonkwo Pellegrini Quarles Rasmussen Sandoval Thackeray
    Underhill Vasquez Whitlock Yarborough Zielinski Abernethy Bellamy Cavendish Delacroix Everhart
    Fontaine Galloway Hargrove Iversen Jorgensen Kowalczyk Lockhart Montague Novak Ostrander
    Prescott Quimby Rutherford Sinclair Tavares Ulloa Vandermeer Wexford Yoshida Zimmerman
    Albrecht Blackwood Calloway Draper Ellsworth Gallagher Holloway Ismail Jacobsen Kendrick
    Lavigne Mortensen Nakamura Oyelaran Penhaligon Stavros Valentine Whitaker Yeoman Amundsen
    Birkeland Corrigan Dalgleish Engstrom Farnsworth Grunewald Haverford Ibarra Jankowski Kilpatrick
    Lorentzen Mackenzie Nightingale Oberholtzer Pargeter Quennell Rosenthal Strand Trevelyan
    Uppington Vickery Wetherell Yancey Zaleski Aldridge Brightwell Cranmer Dellinger Eastwood
    Gundersen Hatfield Irvine Jessop Kincaid Lundgren Merriweather Northcott Ormsby Plimpton
    Quigley Radnor Seward Tolliver Urquhart Vanterpool Winslow Yardley Zeller Ackerley
    Bannister Chadwick Dryden Elwood Farrow Gresham Hollis Inglis Jarrett Keswick Langley
    Marlowe Nesbitt Osgood Pickering Radley Stanhope Thornbury Upshaw Verity Wainwright
    Yelland Zachary Ashby Brough Caldwell Dunham Erskine Foxley Garner Hatherley
    """.split()
)

ORG_HEADS: tuple[str, ...] = tuple(
    """
    Alder Bluestone Cobalt Driftwood Ember Foxglove Granite Harbor Ironbridge Juniper Kestrel
    Lantern Meridian Northwind Obsidian Pinnacle Quarry Redwood Silverline Tidewater Umber Vantage
    Willow Yarrow Zenith Arbor Beacon Crestline Dovetail Evergreen Flintlock Greenfield Highland
    Indigo Jetstream Keystone Lakeside Mapleton Newbridge Oakridge Parkview Quicksilver Riverside
    Stonebridge Thornfield Upland Valleyview Westbrook Yewtree Amberley Bayfield Copperfield
    Deepwater Eastgate Fairhaven Goldcrest Hawthorn Ivybridge Jadestone Kingfisher Larkspur
    Moonstone Nightjar Orchard Pioneer Quillon Rosewood Sandstone Trailhead Ultramarine
    Vineyard Wildflower Yellowwood Zephyrus Ambrose Brightside Cloudbreak Dunstable Everglade
    Firebrand Glenmore Heatherfield Ironwood Jasperwood Knightsbridge Lighthouse Millstone
    Northgate Overland Pennywell Quartzite Ravenscroft Starling Thistledown Underwood
    """.split()
)

LOCATION_NAMES: tuple[str, ...] = tuple(
    """
    Brindle Carrowmere Dunhallow Elmstead Fernhollow Gravesend Hartwell Ilkeston Jarrowby
    Kettleby Lowmoor Marshfield Netherby Oldcastle Pennard Queensferry Rushmoor Saltmarsh
    Tarrant Ulverby Violet Wrenfield Yarmouth Zennor Aldermoor Brackenridge Coldwater Daleford
    Eskdale Fallowfield Glenrock Hobsbay Inverlea Juniperhill Kirkwall Lindale Mossgiel Norwood
    Oakhaven Pinewick Quarrydale Ravenglass Stoneleigh Thornley Ashcombe Upperton Vale
    Westerham Yewdale Ambleside Barrowmoor Cragmere Dunnington Eastwick Foxhill Greystones
    Harrowgate Kingsmere Longmeadow Mistley Needham Ormskirk Pendleton Redcliffe Skelton
    """.split()
)

EVENT_HEADS: tuple[str, ...] = tuple(
    """
    Founders Midwinter Springtide Harvest Solstice Equinox Pathfinder Landmark Horizon Cornerstone
    Crossroads Milestone Overture Renaissance Symposium Gathering Homecoming Jubilee Moonrise
    Sunrise Twilight Daybreak Heritage Frontier Voyager Navigator Catalyst Ascent Summitry
    Confluence Reunion Convergence Outlook Perspective Dialogue Exchange Assembly Colloquy
    Tribute Legacy Endeavor Venture Expedition Odyssey Quest Pilgrimage Procession Caravan
    Armistice Parley
    """.split()
)

PRODUCT_HEADS: tuple[str, ...] = tuple(
    """
    Vector Atlas Nimbus Quasar Helix Pulsar Orbit Zephyr Falcon Comet Aurora Boreal Cascade Delta
    Echo Fusion Galaxy Pathway Ion Jupiter Kinetic Lumen Matrix Nebula Onyx Photon Quantum
    Radian Sigma Titan Umbra Vertex Wavelength Xenon Yonder Zenon Apex Bolt Cipher Dynamo
    Element Flux Gradient Harmonic Impulse Jolt Kernel Lattice Monolith Nova
    """.split()
)

NORP_NAMES: tuple[str, ...] = tuple(
    """
    Brindlese Carrovian Dunhallic Elmsteadish Fernish Hartwellian Kettlebian Lowmoorish
    Marshian Netherbian Pennardic Rushmorian Saltish Tarrantine Ulverbian Wrenfieldian
    Aldermoorish Brackenian Coldwatric Dalefordian Eskdalian Glenrockish Inverlean Mossgielan
    Norwoodish
    """.split()
)

# Words the generator already uses, which no name may shadow.
_RESERVED = frozenset(
    {
        "holdings", "group", "labs", "industries", "partners", "systems", "works", "summit",
        "accord", "talks", "expo", "conference", "festival", "forum", "act", "greater", "the",
        "of", "and", "inc", "corp", "company", "person", "organization", "location", "event",
        "product", "ia", "ton", "burg", "ford", "mere", "ian", "ese", "ish",
    }
)  # fmt: skip


# --- composed names ---------------------------------------------------------------------------
# The curated names above are the first choices. Past them a type draws from words built from a
# stem and an ending (``ash`` + ``worth``), each family with endings of its own so two families
# cannot make the same word. A built word is dropped if it repeats any other word in any pool or
# is a prefix of one, so the finished pools meet the rules in ``check_bank``.

_SURNAME_STEMS = tuple(
    """
    ash bar bax bec bell ben bir blan bol bram bran bren bri bro bry buck bur but cal cam car cas
    cat chal cham chan char chat che chi cla cle clif clin col con cor cot cra cro cul cum dal dan
    dar dav dea del den der dew dil dod dor dow dra dun dur eag ear eck edd eld ell elm emb eng eve
    fai fal far fen fer fin fis fla fle fol for fos fox fra fre fry ful gai gal gar gas gat gil
    gla glo god gol gra gre gri gro gun hab had hal ham han har has hat hav haw hay hea hel hen
    her hew hig hil hob hod hol hop hor hou how hub hud hug hun hur hut
    """.split()
)
_SURNAME_ENDINGS = tuple(
    """
    worth ley well hurst wick stead by thorpe combe holme shaw wold den ham low mont vale brook
    field wood more son man ridge wright smith stone land
    """.split()
)
_PLACE_ENDINGS = tuple(
    """
    ton ford mere moor dale haven bury port minster cliff burn gate mouth stow chester caster
    thwaite side borough wich
    """.split()
)
_ORG_MODS = tuple(
    """
    Blue Red Green Grey Silver Golden Iron Stone Oak Elm Pine Cedar Maple Birch Willow Amber Coral
    Ivory Jade Crimson Scarlet Violet Copper Bronze Steel Crystal Granite Marble North South East
    West Upper Lower Old New Bright Clear Swift Bold Noble Royal Grand High Deep Wild Quiet
    Hollow Mossy Sunny Misty Frosty Windy Rocky Sandy Stormy Silent Gentle Lucky Proud Humble
    Honest Merry Cosy Lofty Rural Coastal Alpine Polar Solar Lunar Stellar Cosmic Rapid Steady
    Vivid Mellow
    """.split()
)
_ORG_NOUNS = tuple(
    """
    bridge field gate wood water stone ridge brook hill vale haven port cliff lake river bay shore
    peak crest point grove meadow orchard garden mill harbor island heath moor spring fall
    crossing landing quarry summit terrace court row lane square brae glen dell wharf quay pier
    tower castle abbey chapel manor barn rock sands dunes falls rapids cove inlet reef ledge bluff
    mesa
    """.split()
)
_EVENT_STEMS = tuple(
    """
    sun moon star dawn dusk day night spring summer autumn winter harvest frost snow rain wind
    storm tide river forest mountain valley sky cloud fire ember aurora comet meadow lake ocean
    shore canyon prairie glacier orchard garden lantern bonfire horizon
    """.split()
)
_EVENT_ENDINGS = tuple(
    """
    light rise fall tide wave line beam glow watch crest song dance feast fair gate quest walk
    rush shine bloom gathering call chime drift flare march sweep dream rite
    """.split()
)
_PRODUCT_STEMS = tuple(
    """
    aero hydro nano opti veri syn tera vita lumi flexi proto omni hexa octa terra volt cryo bio
    geo luma nexa zeta alto pyro astro cyber dyna endo flora gyro halo infra juno kilo lexi mega
    neo orbi para quanta radi sono tela ultra vera xylo yotta zenit alpha beta gamma
    """.split()
)
_PRODUCT_ENDINGS = tuple(
    """
    core link flow grid forge scope pulse node beam wave drive lab stream shift field path gear
    mesh bridge frame mark sync dock vault
    """.split()
)

_FIRST_NAMES_MORE: tuple[str, ...] = tuple(
    """
    Adela Barnaby Cecil Dorothea Eamon Freya Gwen Harlan Irene Jules Klara Leopold Magda Nolan
    Ophelia Percy Quincy Rhea Silas Thea Ursula Vince Winona Xavi Yolanda Zeke Abel Blythe Cyrus
    Dahlia Elias Flora Gavin Honor Imogen Jarek Kira Lorcan Maeve Nils Opal Pierce Rhys Sasha
    Tamsin Ugo Vesna Wren Yvette Zelda Alaric Brigid Casimir Davina Ewan Fenella Gareth Hester
    Inigo Jocelyn Kieran Lysander Marlo Neve Osric Philippa Ruben Saskia Teague Ulla Vaughn Wilma
    Yves Zinnia Briar Cato Doreen Ezra Fern Gilda Ilse Jonty Kendra Lars Mabel Nico
    Olwen Pim Ronan Stella Tove Valda Wystan Yoel Zaria Alwyn Bram Cleo Dov Ellis Fabia Greer
    Hale Indra Jory Kenzo Luz Merrin Niall Orson
    """.split()
)


def _build() -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """The finished pools: curated names first, then built ones that break no rule."""
    first_all = (*FIRST_NAMES, *_FIRST_NAMES_MORE)
    # Every curated word is reserved before anything is built, so a built word can never take one.
    taken: set[str] = {
        w.casefold()
        for pool in (
            first_all,
            SURNAMES,
            ORG_HEADS,
            LOCATION_NAMES,
            EVENT_HEADS,
            PRODUCT_HEADS,
            NORP_NAMES,
        )
        for w in pool
    }

    def take(curated: tuple[str, ...], built: list[str]) -> list[str]:
        """``curated`` plus the built words that are new and extend no word of this pool."""
        fresh = []
        for word in built:
            key = word.casefold()
            if key not in taken and key not in _RESERVED:
                taken.add(key)
                fresh.append(word)
        # A word that is a prefix of another is a short form of it. The curated ones stay; a built
        # word goes if it extends any shorter word, or is itself extended.
        everything = sorted({w.casefold() for w in (*curated, *fresh)})
        prefixes = {a for a, b in zip(everything, everything[1:], strict=False) if b.startswith(a)}
        built_ok = [
            w
            for w in fresh
            if w.casefold() not in prefixes
            and not any(w.casefold().startswith(p) for p in prefixes)
        ]
        taken.difference_update(w.casefold() for w in fresh if w not in built_ok)
        return [*curated, *built_ok]

    names = take(first_all, [])
    surnames = take(
        SURNAMES, [s.capitalize() + e for s in _SURNAME_STEMS for e in _SURNAME_ENDINGS]
    )
    orgs = take(ORG_HEADS, [m + n for m in _ORG_MODS for n in _ORG_NOUNS if m.lower() != n])
    places = take(
        LOCATION_NAMES, [s.capitalize() + e for s in _SURNAME_STEMS for e in _PLACE_ENDINGS]
    )
    events = take(EVENT_HEADS, [s.capitalize() + e for s in _EVENT_STEMS for e in _EVENT_ENDINGS])
    products = take(
        PRODUCT_HEADS, [s.capitalize() + e for s in _PRODUCT_STEMS for e in _PRODUCT_ENDINGS]
    )
    # A demonym is a place with an ending: Carrowmere, Carrowmerian. Only the built ones are new.
    suffixes = ("ian", "ese", "ish")
    norps = take(
        NORP_NAMES,
        [
            (p[:-1] if p.endswith("e") else p) + suffixes[i % len(suffixes)]
            for i, p in enumerate(places)
        ],
    )
    pools = {
        "PERSON": tuple(surnames),
        "ORG": tuple(orgs),
        "LOCATION": tuple(places),
        "EVENT": tuple(events),
        "PRODUCT": tuple(products),
        "NORP": tuple(norps),
    }
    return pools, tuple(names)


POOLS, FIRST_NAMES_ALL = _build()

#: How many of each pool's first words are curated; the rest are built. A world draws the curated
#: ones first, so a small world reads like real names and only a large one reaches the built ones.
CURATED: dict[str, int] = {
    "PERSON": len(SURNAMES),
    "ORG": len(ORG_HEADS),
    "LOCATION": len(LOCATION_NAMES),
    "EVENT": len(EVENT_HEADS),
    "PRODUCT": len(PRODUCT_HEADS),
    "NORP": len(NORP_NAMES),
}

#: How many names of each type a world can draw. A person needs a first name and a surname; the
#: surname is what must stay unique, so the surnames bound the people.
CAPACITY: dict[str, int] = {entity_type: len(pool) for entity_type, pool in POOLS.items()}


def check_bank() -> list[str]:
    """The ways the bank could make two entities collide by accident; empty means it is clean.

    The rules, each of which exists because a name is turned into aliases by one word of it:

    * every name is one capitalised word of letters (or letters and digits for a product);
    * every word is unique across all pools, so no surname, head or place can double as another
      entity's alias, and first names are disjoint from every other pool;
    * no word is a word the generator adds itself (a suffix, a type label, "Greater");
    * no word is a plain prefix of another within the same pool, which would make one a short form
      of the other.
    """
    problems: list[str] = []
    seen: dict[str, str] = {}
    pools = {**POOLS, "FIRST": FIRST_NAMES_ALL}
    for label, pool in pools.items():
        if len(pool) != len(set(w.casefold() for w in pool)):
            problems.append(f"{label} repeats a word")
        for word in pool:
            if not (word[:1].isupper() and word.isalnum() and word.isascii()):
                problems.append(f"{label} {word!r} is not one capitalised word")
            key = word.casefold()
            if key in _RESERVED:
                problems.append(f"{label} {word!r} is a word the generator uses")
            if key in seen and seen[key] != label:
                problems.append(f"{word!r} is in both {seen[key]} and {label}")
            seen.setdefault(key, label)
        lowered = sorted(w.casefold() for w in pool)
        for a, b in zip(lowered, lowered[1:], strict=False):
            if b.startswith(a):
                problems.append(f"{label}: {a!r} is a prefix of {b!r}")
    return problems
