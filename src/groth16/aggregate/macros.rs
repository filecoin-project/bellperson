macro_rules! try_par {
    ($(let $name:ident = $f:expr),+) => {
        $(
            let mut $name = None;
        )+
            rayon::scope(|s| {
                $(
                    let $name = &mut $name;
                    s.spawn(move |_| {
                        *$name = Some($f);
                    });)+
            });
        $(
            let $name = $name.unwrap()?;
        )+
    };
}

macro_rules! par {
    ($(let $name:ident = $f:expr),+) => {
        $(
            let mut $name = None;
        )+
            rayon::scope(|s| {
                $(
                    let $name = &mut $name;
                    s.spawn(move |_| {
                        *$name = Some($f);
                    });)+
            });
        $(
            let $name = $name.unwrap();
        )+
    };

    ($(let ($name1:ident, $name2:ident) = $f:block),+) => {
        $(
            let mut $name1 = None;
            let mut $name2 = None;
        )+
            rayon::scope(|s| {
                $(
                    let $name1 = &mut $name1;
                    let $name2 = &mut $name2;
                    s.spawn(move |_| {
                        let (a, b) = $f;
                        *$name1 = Some(a);
                        *$name2 = Some(b);
                    });)+
            });
        $(
            let $name1 = $name1.unwrap();
            let $name2 = $name2.unwrap();
        )+
    }
}

macro_rules! mul {
    ($a:expr, $b:expr) => {{
        let mut a = $a;
        a.mul_assign($b);
        a
    }};
}

macro_rules! add {
    ($a:expr, $b:expr) => {{
        let mut a = $a;
        a.add_assign($b);
        a
    }};
}

macro_rules! sub {
    ($a:expr, $b:expr) => {{
        let mut a = $a;
        a.sub_assign($b);
        a
    }};
}
